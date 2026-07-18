"""SQLite-backed worker integration tests with offline model fakes."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

from personal_rag.config import Settings
from personal_rag.database import Database
from personal_rag.errors import RagError
from personal_rag.job_service import JobProcessor, classify_job_failure
from personal_rag.models import (
    DocumentStatus,
    JobKind,
    JobRecord,
    JobStage,
    JobStatus,
    UploadReceipt,
)
from personal_rag.parsers import DocumentParser
from personal_rag.rag_service import RAGService
from personal_rag.repository import Repository
from personal_rag.vector_store import VectorStore
from personal_rag.worker import Worker
from scripts.backup import create_backup
from scripts.restore import restore_backup
from tests.fakes import DeterministicEmbedding, FakeLLM


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs: int) -> None:
        self.value += timedelta(**kwargs)


class FlakyParser:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, path: Path, display_name: str | None = None) -> list[object]:
        del path, display_name
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("offline simulated timeout")
        return [{"text": "The Atlas launch key is cobalt blue.", "section": "Atlas"}]


class FailingDeleteRAG:
    def delete_document(self, document_id: str) -> int:
        del document_id
        raise RagError(
            "vector_delete_incomplete",
            "The vector backend did not confirm complete document deletion.",
            status_code=503,
            retryable=True,
        )


class BlockingProcessor:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def process(self, job: JobRecord) -> None:
        del job
        self.started.set()
        self.release.wait(timeout=2)


class HeartbeatRepository:
    def __init__(self, job: JobRecord) -> None:
        self.job = job
        self.leased = False
        self.job_heartbeat_seen = threading.Event()
        self.worker_heartbeats = 0

    def lease_next_job(
        self,
        worker_id: str,
        *,
        lease_seconds: int | None = None,
        kinds: object = None,
    ) -> JobRecord | None:
        del worker_id, lease_seconds, kinds
        if self.leased:
            return None
        self.leased = True
        return self.job

    def record_worker_heartbeat(self, worker_id: str) -> object:
        del worker_id
        self.worker_heartbeats += 1
        return object()

    def heartbeat_job(
        self,
        job_id: str,
        worker_id: str,
        *,
        lease_seconds: int | None = None,
    ) -> JobRecord:
        del job_id, worker_id, lease_seconds
        self.job_heartbeat_seen.set()
        return self.job


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        auth_enabled=False,
        data_dir=tmp_path,
        qdrant_mode="memory",
        qdrant_collection="worker_test_collection",
        embedding_dimensions=256,
        chunk_size=128,
        chunk_overlap=16,
        job_max_attempts=3,
        job_lease_seconds=10,
        worker_poll_seconds=0.1,
    )


def make_repository(
    settings: Settings,
    *,
    max_attempts: int = 3,
    clock: MutableClock | None = None,
) -> Repository:
    repository = Repository(
        Database(settings.database_path),
        lease_seconds=settings.job_lease_seconds,
        max_attempts=max_attempts,
        clock=clock or (lambda: datetime.now(UTC)),
    )
    repository.initialize()
    return repository


def queue_markdown(
    settings: Settings,
    repository: Repository,
    *,
    document_id: str = "doc-atlas",
    content_sha256: str = "a" * 64,
) -> UploadReceipt:
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    path = settings.uploads_dir / f"{document_id}.md"
    path.write_text(
        "# Atlas\n\nThe Atlas launch key is cobalt blue.\n",
        encoding="utf-8",
    )
    return repository.create_document_with_job(
        document_id=document_id,
        display_name="knowledge_base.md",
        stored_path=path.name,
        content_type="text/markdown",
        extension=".md",
        content_sha256=content_sha256,
        size_bytes=path.stat().st_size,
        embedding_fingerprint=settings.embedding_profile.fingerprint,
    )


def make_rag(settings: Settings, repository: Repository) -> tuple[RAGService, VectorStore]:
    store = VectorStore(settings)
    service = RAGService(
        settings,
        store,
        DeterministicEmbedding(settings.embedding_dimensions),
        FakeLLM("The Atlas launch key is cobalt blue [S1]."),
        repository,
    )
    return service, store


def test_ingest_job_reaches_ready_and_records_worker_heartbeat(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    repository = make_repository(settings)
    receipt = queue_markdown(settings, repository)
    rag, store = make_rag(settings, repository)
    processor = JobProcessor(settings, repository, DocumentParser(settings), rag)
    worker = Worker(settings, repository, processor, worker_id="worker-ingest")

    assert worker.run_once() is True

    document = repository.get_document(receipt.document.id)
    job = repository.get_job(receipt.job.id)
    assert document is not None
    assert document.status is DocumentStatus.READY
    assert document.active_version == 1
    assert document.chunk_count > 0
    assert store.count(document_id=document.id) == document.chunk_count
    assert job is not None and job.status is JobStatus.SUCCEEDED
    assert repository.read_worker_heartbeat() is not None


def test_profile_migration_reindex_updates_fingerprint_and_future_dedup(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    repository = make_repository(settings)
    receipt = queue_markdown(settings, repository)
    initial_rag, _ = make_rag(settings, repository)
    initial_worker = Worker(
        settings,
        repository,
        JobProcessor(settings, repository, DocumentParser(settings), initial_rag),
        worker_id="worker-initial-profile",
    )
    assert initial_worker.run_once() is True

    migrated_settings = settings.model_copy(
        update={
            "qdrant_collection": "worker_test_collection_v2",
            "chunk_size": 192,
        }
    )
    reindex_job = repository.request_reindex(receipt.document.id)
    migrated_rag, migrated_store = make_rag(migrated_settings, repository)
    migrated_worker = Worker(
        migrated_settings,
        repository,
        JobProcessor(
            migrated_settings,
            repository,
            DocumentParser(migrated_settings),
            migrated_rag,
        ),
        worker_id="worker-migrated-profile",
    )
    assert migrated_worker.run_once() is True
    assert repository.get_job(reindex_job.id).status is JobStatus.SUCCEEDED  # type: ignore[union-attr]

    migrated_document = repository.get_document(receipt.document.id)
    assert migrated_document is not None
    assert (
        migrated_document.embedding_fingerprint == migrated_settings.embedding_profile.fingerprint
    )
    assert migrated_store.count(document_id=receipt.document.id) == migrated_document.chunk_count

    duplicate = repository.create_document_with_job(
        document_id="duplicate-after-migration",
        display_name="knowledge_base-copy.md",
        stored_path="duplicate-after-migration.md",
        content_type="text/markdown",
        extension=".md",
        content_sha256="a" * 64,
        size_bytes=42,
        embedding_fingerprint=migrated_settings.embedding_profile.fingerprint,
        idempotency_key="upload-after-profile-migration",
    )
    assert duplicate.duplicate is True
    assert duplicate.document.id == receipt.document.id


def test_retryable_failure_requeues_then_succeeds(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    repository = make_repository(settings, max_attempts=2)
    receipt = queue_markdown(settings, repository)
    parser = FlakyParser()
    rag, _ = make_rag(settings, repository)
    processor = JobProcessor(settings, repository, parser, rag)
    worker = Worker(settings, repository, processor, worker_id="worker-retry")

    assert worker.run_once() is True
    retrying = repository.get_job(receipt.job.id)
    assert retrying is not None
    assert retrying.status is JobStatus.RETRYING
    assert repository.get_document(receipt.document.id).status is DocumentStatus.QUEUED  # type: ignore[union-attr]

    assert worker.run_once() is True
    completed = repository.get_job(receipt.job.id)
    assert completed is not None and completed.status is JobStatus.SUCCEEDED
    assert completed.attempts == 2
    assert parser.calls == 2


def test_expired_lease_is_reclaimed_by_next_worker(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    clock = MutableClock()
    repository = make_repository(settings, max_attempts=2, clock=clock)
    receipt = queue_markdown(settings, repository)
    crashed_lease = repository.lease_next_job(
        "worker-crashed", lease_seconds=settings.job_lease_seconds
    )
    assert crashed_lease is not None
    clock.advance(seconds=settings.job_lease_seconds + 1)
    rag, _ = make_rag(settings, repository)
    processor = JobProcessor(settings, repository, DocumentParser(settings), rag)
    replacement = Worker(settings, repository, processor, worker_id="worker-replacement")

    assert replacement.run_once() is True

    completed = repository.get_job(receipt.job.id)
    assert completed is not None
    assert completed.status is JobStatus.SUCCEEDED
    assert completed.attempts == 2


def test_delete_failure_is_visible_until_zero_readback_succeeds(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    repository = make_repository(settings, max_attempts=1)
    receipt = queue_markdown(settings, repository)
    queued_document = repository.get_document(receipt.document.id)
    assert queued_document is not None
    stored_path = settings.uploads_dir / queued_document.stored_path
    rag, store = make_rag(settings, repository)
    ingest = Worker(
        settings,
        repository,
        JobProcessor(settings, repository, DocumentParser(settings), rag),
        worker_id="worker-ingest",
    )
    assert ingest.run_once() is True
    delete_job = repository.request_delete(receipt.document.id)
    failing_processor = JobProcessor(
        settings,
        repository,
        DocumentParser(settings),
        FailingDeleteRAG(),  # type: ignore[arg-type]
    )
    failing_worker = Worker(settings, repository, failing_processor, worker_id="worker-delete-fail")

    assert failing_worker.run_once() is True
    failed = repository.get_job(delete_job.id)
    document = repository.get_document(receipt.document.id)
    assert failed is not None and failed.status is JobStatus.FAILED
    assert document is not None and document.status is DocumentStatus.DELETION_FAILED
    assert store.count(document_id=receipt.document.id) > 0
    assert stored_path.exists()

    retry_delete = repository.request_delete(receipt.document.id)
    successful = Worker(
        settings,
        repository,
        JobProcessor(settings, repository, DocumentParser(settings), rag),
        worker_id="worker-delete-success",
    )
    assert successful.run_once() is True
    assert repository.get_job(retry_delete.id).status is JobStatus.SUCCEEDED  # type: ignore[union-attr]
    assert repository.get_document(receipt.document.id) is None
    assert store.count(document_id=receipt.document.id) == 0
    assert stored_path.exists() is False


def test_relocated_restore_reindexes_and_deletes_only_restored_upload(tmp_path: Path) -> None:
    original_settings = make_settings(tmp_path / "original")
    original_repository = make_repository(original_settings)
    receipt = queue_markdown(original_settings, original_repository)
    original_source = original_settings.uploads_dir / "doc-atlas.md"
    with original_repository.database.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE documents SET stored_path = ? WHERE id = ?",
            (str(original_source), receipt.document.id),
        )
    original_rag, _ = make_rag(original_settings, original_repository)
    original_worker = Worker(
        original_settings,
        original_repository,
        JobProcessor(
            original_settings,
            original_repository,
            DocumentParser(original_settings),
            original_rag,
        ),
        worker_id="worker-before-backup",
    )
    assert original_worker.run_once() is True

    archive = create_backup(original_settings.data_dir, tmp_path / "relocatable.tar.gz")
    restored_root = restore_backup(archive, tmp_path / "restored")
    restored_settings = make_settings(restored_root)
    restored_repository = make_repository(restored_settings)
    restored_document = restored_repository.get_document(receipt.document.id)
    assert restored_document is not None
    assert restored_document.stored_path == "doc-atlas.md"
    restored_source = restored_settings.uploads_dir / restored_document.stored_path
    assert restored_source.is_file()

    restored_rag, restored_store = make_rag(restored_settings, restored_repository)
    reindex_job = restored_repository.request_reindex(receipt.document.id)
    assert reindex_job is not None
    reindex_worker = Worker(
        restored_settings,
        restored_repository,
        JobProcessor(
            restored_settings,
            restored_repository,
            DocumentParser(restored_settings),
            restored_rag,
        ),
        worker_id="worker-after-restore",
    )
    assert reindex_worker.run_once() is True
    reindexed = restored_repository.get_document(receipt.document.id)
    assert reindexed is not None
    assert reindexed.status is DocumentStatus.READY
    assert reindexed.active_version == 2
    assert restored_store.count(document_id=receipt.document.id) == reindexed.chunk_count

    delete_job = restored_repository.request_delete(receipt.document.id)
    assert delete_job is not None
    delete_worker = Worker(
        restored_settings,
        restored_repository,
        JobProcessor(
            restored_settings,
            restored_repository,
            DocumentParser(restored_settings),
            restored_rag,
        ),
        worker_id="worker-delete-restored",
    )
    assert delete_worker.run_once() is True
    assert restored_repository.get_document(receipt.document.id) is None
    assert restored_source.exists() is False
    assert original_source.is_file()


def test_worker_stop_event_prevents_new_lease_and_loop_exits(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    repository = make_repository(settings)
    rag, _ = make_rag(settings, repository)
    worker = Worker(
        settings,
        repository,
        JobProcessor(settings, repository, DocumentParser(settings), rag),
        worker_id="worker-stop",
    )
    thread = threading.Thread(target=worker.run_forever)
    thread.start()
    assert repository.read_worker_heartbeat() is not None or thread.is_alive()

    worker.stop()
    thread.join(timeout=2)

    assert thread.is_alive() is False
    assert worker.run_once() is False


def test_worker_keeps_lease_alive_during_long_processing(tmp_path: Path) -> None:
    settings = make_settings(tmp_path).model_copy(update={"job_lease_seconds": 1})
    now = datetime.now(UTC)
    job = JobRecord(
        id="job-heartbeat",
        document_id="doc-heartbeat",
        kind=JobKind.INGEST,
        status=JobStatus.RUNNING,
        stage=JobStage.QUEUED,
        attempts=1,
        max_attempts=3,
        lease_owner="worker-heartbeat",
        lease_expires_at=now + timedelta(seconds=1),
        created_at=now,
        updated_at=now,
    )
    repository = HeartbeatRepository(job)
    processor = BlockingProcessor()
    worker = Worker(
        settings,
        repository,
        processor,  # type: ignore[arg-type]
        worker_id="worker-heartbeat",
    )
    thread = threading.Thread(target=worker.run_once)
    thread.start()
    assert processor.started.wait(timeout=1)

    assert repository.job_heartbeat_seen.wait(timeout=1)
    processor.release.set()
    thread.join(timeout=2)

    assert thread.is_alive() is False
    assert repository.worker_heartbeats >= 2


def test_retry_classifier_keeps_unknown_errors_terminal() -> None:
    retryable = classify_job_failure(TimeoutError("offline timeout"))
    terminal = classify_job_failure(ValueError("do not persist this detail"))

    assert retryable.retryable is True
    assert retryable.code == "dependency_unavailable"
    assert terminal.retryable is False
    assert terminal.code == "job_processing_error"
    assert "do not persist" not in terminal.message
