"""Durable ingest, reindex, and delete job state-machine processing."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from personal_rag.config import Settings
from personal_rag.errors import RagError
from personal_rag.models import (
    DocumentRecord,
    DocumentStatus,
    JobKind,
    JobRecord,
    JobStage,
)
from personal_rag.rag_service import RAGService
from personal_rag.storage import resolve_managed_upload_path


class JobRepository(Protocol):
    """Persistence operations required by the processor."""

    def get_document(self, document_id: str) -> DocumentRecord | None: ...

    def heartbeat_job(
        self,
        job_id: str,
        worker_id: str,
        *,
        lease_seconds: int | None = None,
    ) -> JobRecord: ...

    def update_job_stage(
        self,
        job_id: str,
        worker_id: str,
        stage: JobStage,
        *,
        progress: float,
        document_status: DocumentStatus | None = None,
    ) -> JobRecord: ...

    def complete_job(
        self,
        job_id: str,
        worker_id: str,
        *,
        chunk_count: int | None = None,
        active_version: int | None = None,
        embedding_fingerprint: str | None = None,
    ) -> JobRecord: ...

    def fail_job(
        self,
        job_id: str,
        worker_id: str,
        *,
        error_code: str,
        error_message: str,
        retryable: bool,
    ) -> JobRecord: ...


class DocumentParserProtocol(Protocol):
    def parse(self, path: Path, display_name: str | None = None) -> Sequence[object]: ...


class JobProcessor:
    """Process one leased job and persist every terminal outcome."""

    def __init__(
        self,
        settings: Settings,
        repository: JobRepository,
        parser: DocumentParserProtocol,
        rag_service: RAGService,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.parser = parser
        self.rag_service = rag_service

    def process(self, job: JobRecord) -> None:
        """Run a leased job; failures are normalized and persisted, not swallowed."""

        worker_id = job.lease_owner
        if not worker_id:
            raise ValueError("a job must be leased before it can be processed")
        try:
            document = self.repository.get_document(job.document_id)
            if document is None:
                raise RagError(
                    "document_not_found",
                    "The document for this job no longer exists.",
                    status_code=404,
                )
            if job.kind == JobKind.DELETE:
                self._delete(job, document, worker_id)
            else:
                self._index(job, document, worker_id)
        except Exception as exc:
            failure = classify_job_failure(exc)
            self.repository.fail_job(
                job.id,
                worker_id,
                error_code=failure.code,
                error_message=failure.message,
                retryable=failure.retryable,
            )

    def _index(self, job: JobRecord, document: DocumentRecord, worker_id: str) -> None:
        self.repository.update_job_stage(
            job.id,
            worker_id,
            stage=JobStage.VALIDATING,
            progress=0.05,
            document_status=DocumentStatus.VALIDATING,
        )
        self._heartbeat(job.id, worker_id)
        self.repository.update_job_stage(
            job.id,
            worker_id,
            stage=JobStage.EXTRACTING,
            progress=0.15,
            document_status=DocumentStatus.EXTRACTING,
        )
        sections = self.parser.parse(
            self._validate_stored_file_path(document), document.display_name
        )
        self._heartbeat(job.id, worker_id)
        self.repository.update_job_stage(
            job.id,
            worker_id,
            stage=JobStage.CHUNKING,
            progress=0.30,
            document_status=DocumentStatus.CHUNKING,
        )
        target_version = max(1, document.active_version + 1)
        self.repository.update_job_stage(
            job.id,
            worker_id,
            stage=JobStage.EMBEDDING,
            progress=0.45,
            document_status=DocumentStatus.EMBEDDING,
        )
        chunk_count = self.rag_service.ingest(
            document,
            sections,
            version=target_version,
        )
        self._heartbeat(job.id, worker_id)
        self.repository.update_job_stage(
            job.id,
            worker_id,
            stage=JobStage.INDEXING,
            progress=0.85,
            document_status=DocumentStatus.INDEXING,
        )
        self.repository.update_job_stage(
            job.id,
            worker_id,
            stage=JobStage.VERIFYING,
            progress=0.95,
        )
        self.repository.complete_job(
            job.id,
            worker_id,
            chunk_count=chunk_count,
            active_version=target_version,
            embedding_fingerprint=self.settings.embedding_profile.fingerprint,
        )

    def _delete(self, job: JobRecord, document: DocumentRecord, worker_id: str) -> None:
        self.repository.update_job_stage(
            job.id,
            worker_id,
            stage=JobStage.DELETING,
            progress=0.20,
            document_status=DocumentStatus.DELETING,
        )
        self._heartbeat(job.id, worker_id)
        self._validate_stored_file_path(document)
        self.rag_service.delete_document(job.document_id)
        self._remove_stored_file(document)
        self._heartbeat(job.id, worker_id)
        # Repository completion also removes persisted turns that retain cited
        # content before the document can become privacy-complete.
        self.repository.complete_job(job.id, worker_id)

    def _heartbeat(self, job_id: str, worker_id: str) -> None:
        self.repository.heartbeat_job(
            job_id,
            worker_id,
            lease_seconds=self.settings.job_lease_seconds,
        )

    def _remove_stored_file(self, document: DocumentRecord) -> None:
        """Remove only a file inside the configured upload directory."""

        stored_path = self._validate_stored_file_path(document)
        try:
            stored_path.unlink(missing_ok=True)
        except OSError as exc:
            raise RagError(
                "stored_file_delete_failed",
                "The stored upload could not be removed.",
                status_code=503,
                retryable=True,
            ) from exc

    def _validate_stored_file_path(self, document: DocumentRecord) -> Path:
        """Resolve a managed upload path without authorizing arbitrary deletion."""

        return resolve_managed_upload_path(
            self.settings.uploads_dir,
            document.stored_path,
            document_id=document.id,
            extension=document.extension,
        )


class JobFailure:
    """A safe, persistence-ready failure classification."""

    __slots__ = ("code", "message", "retryable")

    def __init__(self, code: str, message: str, retryable: bool) -> None:
        self.code = code
        self.message = message
        self.retryable = retryable


def classify_job_failure(exc: Exception) -> JobFailure:
    """Classify expected provider/backend failures without persisting raw secrets."""

    if isinstance(exc, RagError):
        return JobFailure(exc.code, exc.message, exc.retryable)
    if isinstance(exc, FileNotFoundError):
        return JobFailure(
            "stored_file_missing",
            "The stored source file is missing and must be uploaded again.",
            False,
        )
    if isinstance(exc, PermissionError):
        return JobFailure(
            "stored_file_unreadable",
            "The stored source file cannot be read.",
            False,
        )
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return JobFailure(
            "dependency_unavailable",
            "A required dependency is temporarily unavailable.",
            True,
        )
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
        return JobFailure(
            "dependency_unavailable",
            "A required dependency is temporarily unavailable.",
            True,
        )
    name = type(exc).__name__.lower()
    if any(token in name for token in ("timeout", "connection", "ratelimit", "locked")):
        return JobFailure(
            "dependency_unavailable",
            "A required dependency is temporarily unavailable.",
            True,
        )
    return JobFailure(
        "job_processing_error",
        "The document job could not be completed.",
        False,
    )
