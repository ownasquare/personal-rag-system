from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from personal_rag.database import SCHEMA_VERSION, Database
from personal_rag.errors import RagError
from personal_rag.models import DocumentStatus, JobKind, JobStage, JobStatus
from personal_rag.repository import Repository


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs: int) -> None:
        self.value += timedelta(**kwargs)


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock()


@pytest.fixture
def repository(tmp_path: Path, clock: MutableClock) -> Repository:
    repo = Repository(
        Database(tmp_path / "state.sqlite3"),
        lease_seconds=30,
        max_attempts=2,
        clock=clock,
    )
    repo.initialize()
    return repo


def create_upload(
    repository: Repository,
    *,
    document_id: str = "doc-1",
    content_sha256: str = "a" * 64,
    embedding_fingerprint: str = "b" * 64,
    idempotency_key: str | None = "upload-1",
):
    return repository.create_document_with_job(
        document_id=document_id,
        display_name="Notes.md",
        stored_path=f"{document_id}.md",
        content_type="text/markdown",
        extension=".md",
        content_sha256=content_sha256,
        size_bytes=42,
        embedding_fingerprint=embedding_fingerprint,
        idempotency_key=idempotency_key,
    )


def test_database_initializes_wal_foreign_keys_and_schema(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    database.initialize()

    with database.connection() as connection:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert mode.lower() == "wal"
    assert foreign_keys == 1
    assert version == SCHEMA_VERSION
    assert {"documents", "jobs", "meta", "upload_idempotency"} <= tables


def test_repository_rejects_runtime_bound_absolute_storage_path(
    repository: Repository,
) -> None:
    with pytest.raises(ValueError, match="relative managed upload key"):
        repository.create_document_with_job(
            document_id="doc-absolute",
            display_name="Notes.md",
            stored_path="/old/runtime/uploads/doc-absolute.md",
            content_type="text/markdown",
            extension=".md",
            content_sha256="a" * 64,
            size_bytes=42,
            embedding_fingerprint="b" * 64,
        )


def test_create_upload_is_atomic_deduplicated_and_idempotent(repository: Repository) -> None:
    first = create_upload(repository)
    retried = create_upload(repository, document_id="ignored-on-retry")
    duplicate = create_upload(
        repository,
        document_id="ignored-on-dedup",
        idempotency_key="upload-2",
    )

    assert first.duplicate is False
    assert retried.duplicate is True
    assert duplicate.duplicate is True
    assert first.document.id == retried.document.id == duplicate.document.id
    assert first.job.id == retried.job.id == duplicate.job.id
    assert repository.count_documents() == 1
    assert repository.count_jobs() == 1

    with pytest.raises(RagError) as error:
        create_upload(
            repository,
            document_id="doc-other",
            content_sha256="c" * 64,
            idempotency_key="upload-1",
        )

    assert error.value.code == "idempotency_conflict"
    assert error.value.status_code == 409


def test_embedding_profile_is_part_of_document_deduplication(repository: Repository) -> None:
    create_upload(repository)
    second = create_upload(
        repository,
        document_id="doc-2",
        embedding_fingerprint="c" * 64,
        idempotency_key="upload-2",
    )

    assert second.duplicate is False
    assert second.document.id == "doc-2"
    assert repository.count_documents() == 2


def test_concurrent_uploads_converge_on_one_document(repository: Repository) -> None:
    barrier = threading.Barrier(3)

    def upload(index: int):
        barrier.wait()
        return create_upload(
            repository,
            document_id=f"concurrent-{index}",
            idempotency_key=f"concurrent-upload-{index}",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(upload, index) for index in range(2)]
        barrier.wait()
        receipts = [future.result(timeout=5) for future in futures]

    assert receipts[0].document.id == receipts[1].document.id
    assert sorted(receipt.duplicate for receipt in receipts) == [False, True]
    assert repository.count_documents() == 1
    assert repository.count_jobs() == 1


def test_document_and_job_listing_are_paginated_and_counted(repository: Repository) -> None:
    for index in range(4):
        create_upload(
            repository,
            document_id=f"doc-{index}",
            content_sha256=f"{index + 1:064x}",
            idempotency_key=f"upload-{index}",
        )

    documents = repository.list_documents(limit=2, offset=1)
    jobs = repository.list_jobs(limit=2, offset=1, status=JobStatus.QUEUED)

    assert len(documents) == 2
    assert len(jobs) == 2
    assert repository.count_documents(status=DocumentStatus.QUEUED) == 4
    assert repository.count_jobs(status=JobStatus.QUEUED) == 4
    assert repository.get_statistics() == {
        "document_count": 4,
        "ready_document_count": 0,
        "chunk_count": 0,
        "queued_job_count": 4,
    }

    with pytest.raises(ValueError, match="limit"):
        repository.list_documents(limit=0)
    with pytest.raises(ValueError, match="offset"):
        repository.list_jobs(offset=-1)


def test_lease_heartbeat_stage_and_completion_are_owner_safe(
    repository: Repository, clock: MutableClock
) -> None:
    receipt = create_upload(repository)
    leased = repository.lease_next_job("worker-a")

    assert leased is not None
    assert leased.id == receipt.job.id
    assert leased.status is JobStatus.RUNNING
    assert leased.attempts == 1

    with pytest.raises(RagError) as wrong_owner:
        repository.heartbeat_job(leased.id, "worker-b")
    assert wrong_owner.value.code == "job_lease_conflict"

    first_expiry = leased.lease_expires_at
    clock.advance(seconds=5)
    heartbeat = repository.heartbeat_job(leased.id, "worker-a", lease_seconds=60)
    assert heartbeat.lease_expires_at is not None
    assert first_expiry is not None
    assert heartbeat.lease_expires_at > first_expiry

    repository.update_job_stage(
        leased.id,
        "worker-a",
        JobStage.VALIDATING,
        progress=0.1,
        document_status=DocumentStatus.VALIDATING,
    )
    repository.update_job_stage(
        leased.id,
        "worker-a",
        JobStage.EXTRACTING,
        progress=0.3,
        document_status=DocumentStatus.EXTRACTING,
    )

    with pytest.raises(RagError) as backwards:
        repository.update_job_stage(
            leased.id,
            "worker-a",
            JobStage.VALIDATING,
            progress=0.2,
        )
    assert backwards.value.code == "invalid_job_transition"

    for stage, progress, status in (
        (JobStage.CHUNKING, 0.5, DocumentStatus.CHUNKING),
        (JobStage.EMBEDDING, 0.7, DocumentStatus.EMBEDDING),
        (JobStage.INDEXING, 0.9, DocumentStatus.INDEXING),
        (JobStage.VERIFYING, 0.95, None),
    ):
        repository.update_job_stage(
            leased.id,
            "worker-a",
            stage,
            progress=progress,
            document_status=status,
        )

    completed = repository.complete_job(
        leased.id,
        "worker-a",
        chunk_count=7,
        active_version=1,
        embedding_fingerprint="c" * 64,
    )
    document = repository.get_document(receipt.document.id)

    assert completed.status is JobStatus.SUCCEEDED
    assert completed.stage is JobStage.COMPLETE
    assert completed.lease_owner is None
    assert document is not None
    assert document.status is DocumentStatus.READY
    assert document.chunk_count == 7
    assert document.active_version == 1
    assert document.embedding_fingerprint == "c" * 64
    duplicate_after_profile_update = create_upload(
        repository,
        document_id="duplicate-after-profile-update",
        embedding_fingerprint="c" * 64,
        idempotency_key="upload-after-profile-update",
    )
    assert duplicate_after_profile_update.duplicate is True
    assert duplicate_after_profile_update.document.id == receipt.document.id
    assert repository.get_statistics() == {
        "document_count": 1,
        "ready_document_count": 1,
        "chunk_count": 7,
        "queued_job_count": 0,
    }


def test_concurrent_workers_cannot_lease_the_same_job(repository: Repository) -> None:
    receipt = create_upload(repository)
    barrier = threading.Barrier(3)

    def lease(worker_id: str):
        barrier.wait()
        return repository.lease_next_job(worker_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(lease, worker_id) for worker_id in ("worker-a", "worker-b")]
        barrier.wait()
        leases = [future.result(timeout=5) for future in futures]

    leased = [job for job in leases if job is not None]
    assert len(leased) == 1
    assert leased[0].id == receipt.job.id


def test_expired_lease_is_reclaimed_then_exhausted(
    repository: Repository, clock: MutableClock
) -> None:
    receipt = create_upload(repository)
    first = repository.lease_next_job("worker-a")
    assert first is not None

    clock.advance(seconds=31)
    second = repository.lease_next_job("worker-b")
    assert second is not None
    assert second.id == first.id
    assert second.attempts == 2
    assert second.lease_owner == "worker-b"

    clock.advance(seconds=31)
    assert repository.lease_next_job("worker-c") is None
    failed_job = repository.get_job(receipt.job.id)
    failed_document = repository.get_document(receipt.document.id)

    assert failed_job is not None
    assert failed_job.status is JobStatus.FAILED
    assert failed_job.error_code == "job_lease_expired"
    assert failed_document is not None
    assert failed_document.status is DocumentStatus.FAILED


def test_retryable_and_terminal_failures_update_document_state(
    repository: Repository,
) -> None:
    receipt = create_upload(repository)
    leased = repository.lease_next_job("worker-a")
    assert leased is not None

    retrying = repository.fail_job(
        leased.id,
        "worker-a",
        error_code="provider_timeout",
        error_message="Embedding provider timed out",
        retryable=True,
    )
    assert retrying.status is JobStatus.RETRYING
    assert retrying.lease_owner is None

    leased_again = repository.lease_next_job("worker-b")
    assert leased_again is not None
    failed = repository.fail_job(
        leased_again.id,
        "worker-b",
        error_code="invalid_document",
        error_message="The document cannot be parsed",
        retryable=False,
    )
    document = repository.get_document(receipt.document.id)

    assert failed.status is JobStatus.FAILED
    assert document is not None
    assert document.status is DocumentStatus.FAILED
    assert document.error_code == "invalid_document"


def test_reindex_and_delete_requests_follow_safe_transitions(repository: Repository) -> None:
    receipt = create_upload(repository)
    leased = repository.lease_next_job("worker-a")
    assert leased is not None
    repository.update_job_stage(
        leased.id,
        "worker-a",
        JobStage.VALIDATING,
        progress=0.2,
        document_status=DocumentStatus.VALIDATING,
    )
    repository.fail_job(
        leased.id,
        "worker-a",
        error_code="bad_input",
        error_message="bad input",
        retryable=False,
    )

    reindex = repository.request_reindex(receipt.document.id)
    assert reindex.kind is JobKind.REINDEX
    assert repository.get_document(receipt.document.id).status is DocumentStatus.REINDEXING  # type: ignore[union-attr]

    delete = repository.request_delete(receipt.document.id)
    assert delete.kind is JobKind.DELETE
    assert repository.get_document(receipt.document.id).status is DocumentStatus.DELETING  # type: ignore[union-attr]

    leased_delete = repository.lease_next_job("worker-delete", kinds=[JobKind.DELETE])
    assert leased_delete is not None
    repository.update_job_stage(
        leased_delete.id,
        "worker-delete",
        JobStage.DELETING,
        progress=0.5,
    )
    repository.complete_job(leased_delete.id, "worker-delete")

    assert repository.get_document(receipt.document.id) is None
    deleted = repository.get_document(receipt.document.id, include_deleted=True)
    assert deleted is not None
    assert deleted.status is DocumentStatus.DELETED
    assert deleted.deleted_at is not None


def test_delete_rejects_a_live_ingest_lease_instead_of_racing_vector_commit(
    repository: Repository,
) -> None:
    receipt = create_upload(repository)
    leased = repository.lease_next_job("worker-ingest")
    assert leased is not None
    repository.update_job_stage(
        leased.id,
        "worker-ingest",
        JobStage.VALIDATING,
        progress=0.1,
        document_status=DocumentStatus.VALIDATING,
    )

    with pytest.raises(RagError) as error:
        repository.request_delete(receipt.document.id)

    assert error.value.code == "document_busy"
    assert error.value.retryable is True
    assert repository.count_jobs(document_id=receipt.document.id) == 1
    persisted_job = repository.get_job(leased.id)
    assert persisted_job is not None
    assert persisted_job.status is JobStatus.RUNNING


def test_ready_document_versions_are_fail_closed_by_state(repository: Repository) -> None:
    receipt = create_upload(repository)
    leased = repository.lease_next_job("worker-ready")
    assert leased is not None
    stages = (
        (JobStage.VALIDATING, DocumentStatus.VALIDATING, 0.1),
        (JobStage.EXTRACTING, DocumentStatus.EXTRACTING, 0.2),
        (JobStage.CHUNKING, DocumentStatus.CHUNKING, 0.3),
        (JobStage.EMBEDDING, DocumentStatus.EMBEDDING, 0.5),
        (JobStage.INDEXING, DocumentStatus.INDEXING, 0.8),
    )
    for stage, document_status, progress in stages:
        repository.update_job_stage(
            leased.id,
            "worker-ready",
            stage,
            progress=progress,
            document_status=document_status,
        )
    repository.update_job_stage(
        leased.id,
        "worker-ready",
        JobStage.VERIFYING,
        progress=0.95,
    )
    repository.complete_job(
        leased.id,
        "worker-ready",
        chunk_count=3,
        active_version=1,
    )

    assert repository.get_ready_document_versions() == {receipt.document.id: 1}
    assert repository.get_ready_document_versions([receipt.document.id, "missing"]) == {
        receipt.document.id: 1
    }

    repository.request_reindex(receipt.document.id)

    assert repository.get_ready_document_versions() == {}


def test_meta_and_worker_heartbeat_round_trip(repository: Repository, clock: MutableClock) -> None:
    assert repository.get_meta("missing") is None
    repository.set_meta("collection_schema", "v1")
    seen_at = repository.record_worker_heartbeat("worker-a")

    assert repository.get_meta("collection_schema") == "v1"
    assert repository.get_meta("worker_last_id") == "worker-a"
    assert repository.read_worker_heartbeat() == seen_at == clock.value


def test_database_rejects_a_newer_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    connection.close()

    with pytest.raises(RuntimeError, match="newer schema"):
        Database(path).initialize()
