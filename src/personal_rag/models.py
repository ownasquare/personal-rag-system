"""Domain and API contracts shared by the service and UI."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"
    REINDEXING = "reindexing"
    DELETING = "deleting"
    DELETION_FAILED = "deletion_failed"
    DELETED = "deleted"


class JobKind(StrEnum):
    INGEST = "ingest"
    REINDEX = "reindex"
    DELETE = "delete"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobStage(StrEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    VERIFYING = "verifying"
    DELETING = "deleting"
    COMPLETE = "complete"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    display_name: str
    stored_path: str = Field(exclude=True, repr=False)
    content_type: str
    extension: str
    content_sha256: str = Field(repr=False)
    size_bytes: int
    status: DocumentStatus
    embedding_fingerprint: str
    active_version: int = 0
    chunk_count: int = 0
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class DocumentPublic(BaseModel):
    id: str
    display_name: str
    content_type: str
    extension: str
    size_bytes: int
    status: DocumentStatus
    active_version: int
    chunk_count: int
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, record: DocumentRecord) -> DocumentPublic:
        return cls.model_validate(
            record.model_dump(exclude={"stored_path", "content_sha256", "embedding_fingerprint"})
        )


class JobRecord(BaseModel):
    id: str
    document_id: str
    kind: JobKind
    status: JobStatus
    stage: JobStage
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    attempts: int = 0
    max_attempts: int = 3
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None


class UploadReceipt(BaseModel):
    document: DocumentPublic
    job: JobRecord
    duplicate: bool = False


class DocumentList(BaseModel):
    items: list[DocumentPublic]
    total: int
    limit: int
    offset: int


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=100)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[str] | None = Field(default=None, max_length=100)


class Citation(BaseModel):
    label: str
    document_id: str
    chunk_id: str
    document_name: str
    page_number: int | None = None
    section: str | None = None
    snippet: str
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    no_answer: bool
    request_id: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool = False
    request_id: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


class DependencyState(BaseModel):
    name: str
    status: Literal["ready", "degraded", "unavailable", "not_configured"]
    detail: str | None = None


class SystemStatus(BaseModel):
    status: Literal["ready", "needs_setup", "degraded"]
    collection: str
    document_count: int
    ready_document_count: int
    chunk_count: int
    queued_job_count: int
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    chat_provider: str
    chat_model: str
    dependencies: list[DependencyState]
    worker_last_seen_at: datetime | None = None
