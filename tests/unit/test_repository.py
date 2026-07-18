from __future__ import annotations

import hashlib
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from personal_rag.database import SCHEMA_VERSION, Database, unicode_casefold
from personal_rag.errors import RagError
from personal_rag.models import (
    Citation,
    ConversationTurnStatus,
    DocumentSort,
    DocumentStatus,
    JobKind,
    JobStage,
    JobStatus,
    SortOrder,
)
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


def create_named_document(
    repository: Repository,
    display_name: str,
    *,
    document_id: str,
    status: DocumentStatus = DocumentStatus.QUEUED,
    extension: str = ".md",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> str:
    """Create one uniquely-backed document and optionally set deterministic list metadata."""

    receipt = repository.create_document_with_job(
        document_id=document_id,
        display_name=display_name,
        stored_path=f"{document_id}{extension}",
        content_type="text/markdown",
        extension=extension,
        content_sha256=hashlib.sha256(document_id.encode()).hexdigest(),
        size_bytes=42,
        embedding_fingerprint="f" * 64,
    )
    if status is not DocumentStatus.QUEUED or created_at is not None or updated_at is not None:
        created = created_at or receipt.document.created_at
        updated = updated_at or created
        with repository.database.transaction(immediate=True) as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = ?, active_version = ?, chunk_count = ?,
                    created_at = ?, updated_at = ?, deleted_at = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    1 if status is DocumentStatus.READY else 0,
                    1 if status is DocumentStatus.READY else 0,
                    created.isoformat(timespec="microseconds"),
                    updated.isoformat(timespec="microseconds"),
                    (
                        updated.isoformat(timespec="microseconds")
                        if status is DocumentStatus.DELETED
                        else None
                    ),
                    document_id,
                ),
            )
    return receipt.document.id


def seed_findability_documents(repository: Repository, *, total: int = 36) -> None:
    """Seed tied sort values across every public library status group."""

    base = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)
    statuses = (
        DocumentStatus.READY,
        DocumentStatus.VALIDATING,
        DocumentStatus.FAILED,
        DocumentStatus.DELETION_FAILED,
        DocumentStatus.DELETED,
        DocumentStatus.QUEUED,
    )
    names = ("Alpha Report.md", "beta report.md", "Résumé Report.md", "Zulu Report.md")
    for index in range(total):
        create_named_document(
            repository,
            names[index % len(names)],
            document_id=f"find-{index:02}",
            status=statuses[index % len(statuses)],
            created_at=base + timedelta(minutes=index // 3),
            updated_at=base + timedelta(minutes=(total - index - 1) // 4),
        )


def mark_document_ready(repository: Repository, document_id: str) -> None:
    """Put a fixture document in authoritative retrieval-ready state."""

    with repository.database.transaction(immediate=True) as connection:
        connection.execute(
            """
            UPDATE documents
            SET status = 'ready', active_version = 1, chunk_count = 1
            WHERE id = ?
            """,
            (document_id,),
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
    assert {
        "documents",
        "jobs",
        "meta",
        "upload_idempotency",
        "conversations",
        "conversation_turns",
        "turn_citations",
    } <= tables


def test_database_registers_deterministic_unicode_casefold_on_every_connection(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()

    with database.connection() as first:
        first_value = first.execute("SELECT unicode_casefold(?)", ("Straße",)).fetchone()[0]
    with database.connection() as second:
        second_value = second.execute("SELECT unicode_casefold(?)", ("\uff32ÉSUMÉ",)).fetchone()[0]

    assert first_value == "strasse"
    assert second_value == "résumé"
    assert unicode_casefold("Straße") == first_value


def test_v1_database_migrates_conversations_without_changing_documents(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    repository = Repository(database)
    receipt = create_upload(repository)

    with database.connection() as connection:
        connection.executescript(
            """
            DROP TABLE turn_citations;
            DROP TABLE conversation_turns;
            DROP TABLE conversations;
            PRAGMA user_version = 1;
            """
        )

    Database(database.path).initialize()

    with database.connection() as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert version == 2
    assert {"conversations", "conversation_turns", "turn_citations"} <= tables
    assert Repository(database).get_document(receipt.document.id) is not None
    assert Repository(database).get_job(receipt.job.id) is not None


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


def test_document_search_is_literal_unicode_and_counted(repository: Repository) -> None:
    create_named_document(
        repository,
        "Résumé 100%_plan.md",
        document_id="doc-special",
        status=DocumentStatus.READY,
    )
    create_named_document(
        repository,
        "Resume 100X-plan.md",
        document_id="doc-near-match",
        status=DocumentStatus.READY,
    )
    create_named_document(
        repository,
        "Straße notes.txt",
        document_id="doc-casefold",
        status=DocumentStatus.READY,
        extension=".txt",
    )
    create_named_document(
        repository,
        "folder/name [draft]*.txt",
        document_id="doc-markdown",
        status=DocumentStatus.READY,
        extension=".txt",
    )
    create_named_document(
        repository,
        "O'Reilly #Roadmap.pdf",
        document_id="doc-quote",
        status=DocumentStatus.READY,
        extension=".pdf",
    )

    assert [item.id for item in repository.list_documents(query="RÉSUMÉ 100%_")] == ["doc-special"]
    assert repository.count_documents(query="RÉSUMÉ 100%_") == 1
    assert [item.id for item in repository.list_documents(query="STRASSE")] == ["doc-casefold"]
    assert [item.id for item in repository.list_documents(query="folder/name")] == ["doc-markdown"]
    assert [item.id for item in repository.list_documents(query="[DRAFT]*")] == ["doc-markdown"]
    assert [item.id for item in repository.list_documents(query="O'REILLY #")] == ["doc-quote"]
    assert [item.id for item in repository.list_documents(query=".PDF")] == ["doc-quote"]


def test_document_search_treats_sql_wildcards_as_literal_text(repository: Repository) -> None:
    create_named_document(repository, "100% complete.md", document_id="doc-percent")
    create_named_document(repository, "100 percent.md", document_id="doc-words")
    create_named_document(repository, "under_score.md", document_id="doc-underscore")

    assert [item.id for item in repository.list_documents(query="%")] == ["doc-percent"]
    assert [item.id for item in repository.list_documents(query="_")] == ["doc-underscore"]


def test_document_status_filters_use_or_semantics_and_preserve_single_status(
    repository: Repository,
) -> None:
    seed_findability_documents(repository)

    attention = repository.list_documents(
        limit=200,
        statuses=[DocumentStatus.FAILED, DocumentStatus.DELETION_FAILED],
    )
    combined = repository.list_documents(
        limit=200,
        status=DocumentStatus.READY,
        statuses=[DocumentStatus.FAILED, DocumentStatus.FAILED],
    )

    assert len(attention) == 12
    assert {item.status for item in attention} == {
        DocumentStatus.FAILED,
        DocumentStatus.DELETION_FAILED,
    }
    assert repository.count_documents(
        statuses=[DocumentStatus.FAILED, DocumentStatus.DELETION_FAILED]
    ) == len(attention)
    assert len(combined) == 12
    assert {item.status for item in combined} == {
        DocumentStatus.READY,
        DocumentStatus.FAILED,
    }
    assert repository.count_documents(status=DocumentStatus.READY) == 6
    assert repository.count_documents(statuses=[]) == 30


@pytest.mark.parametrize("sort", list(DocumentSort))
@pytest.mark.parametrize("order", list(SortOrder))
def test_document_sorts_have_stable_complete_page_boundaries(
    repository: Repository,
    sort: DocumentSort,
    order: SortOrder,
) -> None:
    seed_findability_documents(repository)
    all_items = repository.list_documents(limit=200, sort=sort, order=order)
    reverse = order is SortOrder.DESC
    if sort is DocumentSort.CREATED:
        expected = sorted(all_items, key=lambda item: (item.created_at, item.id), reverse=reverse)
    elif sort is DocumentSort.UPDATED:
        expected = sorted(all_items, key=lambda item: (item.updated_at, item.id), reverse=reverse)
    else:
        expected = sorted(
            all_items,
            key=lambda item: (unicode_casefold(item.display_name), item.id),
            reverse=reverse,
        )

    paged = []
    for offset in range(0, repository.count_documents(), 7):
        paged.extend(
            repository.list_documents(
                limit=7,
                offset=offset,
                sort=sort,
                order=order,
            )
        )

    assert len(all_items) == 30
    assert [item.id for item in all_items] == [item.id for item in expected]
    assert [item.id for item in paged] == [item.id for item in expected]
    assert len({item.id for item in paged}) == len(paged)


def test_document_filters_share_truthful_list_and_count_and_exclude_deleted(
    repository: Repository,
) -> None:
    seed_findability_documents(repository)

    listed = repository.list_documents(
        limit=200,
        query="REPORT",
        statuses=[DocumentStatus.READY, DocumentStatus.FAILED],
        sort=DocumentSort.NAME,
        order=SortOrder.ASC,
    )

    assert len(listed) == 12
    assert repository.count_documents(
        query="REPORT",
        statuses=[DocumentStatus.READY, DocumentStatus.FAILED],
    ) == len(listed)
    assert all(item.status is not DocumentStatus.DELETED for item in listed)
    assert repository.list_documents(query="REPORT", statuses=[DocumentStatus.DELETED]) == []
    assert repository.count_documents(query="REPORT", statuses=[DocumentStatus.DELETED]) == 0
    assert (
        repository.count_documents(
            status=DocumentStatus.DELETED,
            include_deleted=True,
        )
        == 6
    )


@pytest.mark.parametrize("query", ["line\nbreak", "nul\x00byte", "hidden\u200djoiner"])
def test_document_search_rejects_non_visible_characters(
    repository: Repository,
    query: str,
) -> None:
    with pytest.raises(ValueError, match="visible"):
        repository.list_documents(query=query)
    with pytest.raises(ValueError, match="visible"):
        repository.count_documents(query=query)


def test_document_search_and_sort_validation_is_bounded(repository: Repository) -> None:
    with pytest.raises(ValueError, match="200"):
        repository.list_documents(query="x" * 201)
    with pytest.raises(ValueError, match="text"):
        repository.list_documents(query=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        repository.list_documents(sort="random")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        repository.list_documents(order="sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="statuses"):
        repository.list_documents(statuses="ready")  # type: ignore[arg-type]

    assert repository.list_documents(query="   ") == []


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


def test_completed_conversation_turn_is_idempotent_and_round_trips_citations(
    repository: Repository,
) -> None:
    receipt = create_upload(repository)
    mark_document_ready(repository, receipt.document.id)
    conversation = repository.create_conversation()
    citation = Citation(
        label="S1",
        document_id=receipt.document.id,
        chunk_id="chunk-1",
        document_name="Notes.md",
        page_number=2,
        section="Launch",
        snippet="The launch key is cobalt.",
        score=0.91,
    )
    reservation = repository.reserve_conversation_turn(
        conversation.id,
        client_turn_id="client-turn-001",
        question="What is the launch key?",
        top_k=5,
        document_ids=["doc-1"],
        request_fingerprint="a" * 64,
    )
    assert reservation.reservation_token is not None
    completed = repository.complete_conversation_turn(
        reservation.turn.id,
        reservation_token=reservation.reservation_token,
        answer="The launch key is cobalt [S1].",
        citations=[citation],
        no_answer=False,
        request_id="request-1",
    )
    duplicate = repository.reserve_conversation_turn(
        conversation.id,
        client_turn_id="client-turn-001",
        question="What is the launch key?",
        top_k=5,
        document_ids=["doc-1"],
        request_fingerprint="a" * 64,
    )

    assert reservation.created is True
    assert completed.citations == [citation]
    assert duplicate.created is False
    assert duplicate.cached_turn == completed
    assert repository.get_conversation(conversation.id).title == "What is the launch key?"  # type: ignore[union-attr]
    assert repository.count_conversation_turns(conversation.id) == 1


def test_conversation_reservation_conflicts_then_recovers_after_expiry(
    repository: Repository, clock: MutableClock
) -> None:
    conversation = repository.create_conversation("Atlas")
    first = repository.reserve_conversation_turn(
        conversation.id,
        client_turn_id="client-turn-001",
        question="What is the key?",
        top_k=None,
        document_ids=None,
        request_fingerprint="b" * 64,
        reservation_seconds=30,
    )

    with pytest.raises(RagError) as active:
        repository.reserve_conversation_turn(
            conversation.id,
            client_turn_id="client-turn-001",
            question="What is the key?",
            top_k=None,
            document_ids=None,
            request_fingerprint="b" * 64,
            reservation_seconds=30,
        )
    assert active.value.code == "conversation_turn_in_progress"
    assert active.value.retryable is True

    clock.advance(seconds=31)
    recovered = repository.reserve_conversation_turn(
        conversation.id,
        client_turn_id="client-turn-001",
        question="What is the key?",
        top_k=None,
        document_ids=None,
        request_fingerprint="b" * 64,
        reservation_seconds=30,
    )
    assert recovered.created is True
    assert recovered.turn.id == first.turn.id

    failed = repository.fail_conversation_turn(
        recovered.turn.id,
        reservation_token=recovered.reservation_token or "missing-reservation-token",
        error_code="provider_timeout",
        retryable=True,
    )
    assert repository.count_conversation_turns(conversation.id) == 0
    assert repository.count_conversation_turns(conversation.id, include_incomplete=True) == 1
    assert (
        repository.list_conversation_turns(conversation.id, include_incomplete=True)[0].status
        is ConversationTurnStatus.FAILED
    )
    retry = repository.reserve_conversation_turn(
        conversation.id,
        client_turn_id="client-turn-001",
        question="What is the key?",
        top_k=None,
        document_ids=None,
        request_fingerprint="b" * 64,
        reservation_seconds=30,
    )
    assert failed.status is ConversationTurnStatus.FAILED
    assert retry.created is True
    assert retry.turn.status is ConversationTurnStatus.PENDING
    assert retry.turn.id == first.turn.id


def test_reservation_tokens_fence_stale_completion_and_failure(
    repository: Repository, clock: MutableClock
) -> None:
    conversation = repository.create_conversation("Atlas")
    first = repository.reserve_conversation_turn(
        conversation.id,
        client_turn_id="client-turn-001",
        question="What is the key?",
        top_k=None,
        document_ids=None,
        request_fingerprint="f" * 64,
        reservation_seconds=30,
    )
    assert first.reservation_token is not None

    clock.advance(seconds=31)
    recovered = repository.reserve_conversation_turn(
        conversation.id,
        client_turn_id="client-turn-001",
        question="What is the key?",
        top_k=None,
        document_ids=None,
        request_fingerprint="f" * 64,
        reservation_seconds=30,
    )
    assert recovered.reservation_token is not None
    assert recovered.reservation_token != first.reservation_token
    assert (
        repository.renew_conversation_turn_reservation(
            recovered.turn.id,
            reservation_token=first.reservation_token,
            reservation_seconds=30,
        )
        is False
    )
    assert (
        repository.fail_conversation_turn(
            recovered.turn.id,
            reservation_token=first.reservation_token,
            error_code="provider_timeout",
            retryable=True,
        ).status
        is ConversationTurnStatus.PENDING
    )
    with pytest.raises(RagError, match="newer request") as stale_completion:
        repository.complete_conversation_turn(
            recovered.turn.id,
            reservation_token=first.reservation_token,
            answer="Stale answer.",
            citations=[],
            no_answer=True,
            request_id="stale-request",
        )
    assert stale_completion.value.code == "conversation_turn_lease_lost"
    assert repository.renew_conversation_turn_reservation(
        recovered.turn.id,
        reservation_token=recovered.reservation_token,
        reservation_seconds=30,
    )

    completed = repository.complete_conversation_turn(
        recovered.turn.id,
        reservation_token=recovered.reservation_token,
        answer="Current answer.",
        citations=[],
        no_answer=True,
        request_id="current-request",
    )
    after_stale_failure = repository.fail_conversation_turn(
        recovered.turn.id,
        reservation_token=first.reservation_token,
        error_code="provider_timeout",
        retryable=True,
    )
    assert completed.status is ConversationTurnStatus.COMPLETED
    assert after_stale_failure == completed


def test_late_answer_cannot_persist_a_source_that_is_no_longer_ready(
    repository: Repository,
) -> None:
    receipt = create_upload(repository)
    mark_document_ready(repository, receipt.document.id)
    conversation = repository.create_conversation("Atlas")
    reservation = repository.reserve_conversation_turn(
        conversation.id,
        client_turn_id="client-turn-001",
        question="What is the key?",
        top_k=5,
        document_ids=[receipt.document.id],
        request_fingerprint="9" * 64,
    )
    assert reservation.reservation_token is not None
    repository.request_delete(receipt.document.id)

    with pytest.raises(RagError, match="source changed") as changed:
        repository.complete_conversation_turn(
            reservation.turn.id,
            reservation_token=reservation.reservation_token,
            answer="Cobalt [S1].",
            citations=[
                Citation(
                    label="S1",
                    document_id=receipt.document.id,
                    chunk_id="chunk-1",
                    document_name=receipt.document.display_name,
                    snippet="Cobalt.",
                )
            ],
            no_answer=False,
            request_id="late-request",
        )
    assert changed.value.code == "source_changed"
    persisted = repository.get_conversation_turn(reservation.turn.id)
    assert persisted is not None
    assert persisted.status is ConversationTurnStatus.PENDING
    assert persisted.answer is None
    assert persisted.citations == []


def test_conversation_history_is_bounded_and_hard_delete_cascades(
    repository: Repository,
) -> None:
    conversation = repository.create_conversation("Atlas")
    for index in range(3):
        reservation = repository.reserve_conversation_turn(
            conversation.id,
            client_turn_id=f"client-turn-{index:03d}",
            question=f"Question {index}",
            top_k=None,
            document_ids=None,
            request_fingerprint=f"{index + 1:064x}",
        )
        repository.complete_conversation_turn(
            reservation.turn.id,
            reservation_token=reservation.reservation_token or "missing-reservation-token",
            answer=f"Answer {index}",
            citations=[],
            no_answer=True,
            request_id=f"request-{index}",
        )

    history = repository.conversation_history(conversation.id, limit=3)
    assert [(message.role, message.content) for message in history] == [
        ("user", "Question 2"),
        ("assistant", "Answer 2"),
    ]
    assert repository.conversation_history(conversation.id, limit=1) == []
    assert repository.delete_conversation(conversation.id) is True
    assert repository.delete_conversation(conversation.id) is False
    assert repository.get_conversation(conversation.id) is None


def test_document_deletion_purges_whole_cited_turn(repository: Repository) -> None:
    receipt = create_upload(repository)
    mark_document_ready(repository, receipt.document.id)
    conversation = repository.create_conversation("Atlas")
    reservation = repository.reserve_conversation_turn(
        conversation.id,
        client_turn_id="client-turn-001",
        question="What is the key?",
        top_k=5,
        document_ids=[receipt.document.id],
        request_fingerprint="c" * 64,
    )
    repository.complete_conversation_turn(
        reservation.turn.id,
        reservation_token=reservation.reservation_token or "missing-reservation-token",
        answer="Cobalt [S1].",
        citations=[
            Citation(
                label="S1",
                document_id=receipt.document.id,
                chunk_id="chunk-1",
                document_name=receipt.document.display_name,
                snippet="Cobalt.",
            )
        ],
        no_answer=False,
        request_id="request-1",
    )

    delete_job = repository.request_delete(receipt.document.id)
    leased = repository.lease_next_job("worker-delete", kinds=[JobKind.DELETE])
    assert leased is not None and leased.id == delete_job.id
    repository.update_job_stage(
        leased.id,
        "worker-delete",
        JobStage.DELETING,
        progress=0.5,
    )
    repository.complete_job(leased.id, "worker-delete")

    assert repository.count_conversation_turns(conversation.id) == 0
    assert repository.get_conversation(conversation.id) is None


def test_document_deletion_retitles_a_conversation_from_remaining_truth(
    repository: Repository,
) -> None:
    receipt = create_upload(repository)
    mark_document_ready(repository, receipt.document.id)
    conversation = repository.create_conversation()
    cited = repository.reserve_conversation_turn(
        conversation.id,
        client_turn_id="client-turn-001",
        question="Sensitive first question",
        top_k=5,
        document_ids=[receipt.document.id],
        request_fingerprint="d" * 64,
    )
    repository.complete_conversation_turn(
        cited.turn.id,
        reservation_token=cited.reservation_token or "missing-reservation-token",
        answer="Sensitive answer [S1].",
        citations=[
            Citation(
                label="S1",
                document_id=receipt.document.id,
                chunk_id="chunk-1",
                document_name=receipt.document.display_name,
                snippet="Sensitive source text.",
            )
        ],
        no_answer=False,
        request_id="request-1",
    )
    remaining = repository.reserve_conversation_turn(
        conversation.id,
        client_turn_id="client-turn-002",
        question="Remaining question",
        top_k=5,
        document_ids=None,
        request_fingerprint="e" * 64,
    )
    repository.complete_conversation_turn(
        remaining.turn.id,
        reservation_token=remaining.reservation_token or "missing-reservation-token",
        answer="No supported answer.",
        citations=[],
        no_answer=True,
        request_id="request-2",
    )

    delete_job = repository.request_delete(receipt.document.id)
    leased = repository.lease_next_job("worker-delete", kinds=[JobKind.DELETE])
    assert leased is not None and leased.id == delete_job.id
    repository.update_job_stage(
        leased.id,
        "worker-delete",
        JobStage.DELETING,
        progress=0.5,
    )
    repository.complete_job(leased.id, "worker-delete")

    persisted = repository.get_conversation(conversation.id)
    assert persisted is not None
    assert persisted.title == "Remaining question"
    assert persisted.turn_count == 1
    assert [turn.question for turn in repository.list_conversation_turns(conversation.id)] == [
        "Remaining question"
    ]


def test_database_rejects_a_newer_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    connection.close()

    with pytest.raises(RuntimeError, match="newer schema"):
        Database(path).initialize()
